
# Generated: 2025-08-27T20:26:11.262172
# Source Brief: brief_02454.md
# Brief Index: 2454

        
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
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along their path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.FPS = 30
        self.MAX_STEPS = 18000 # 10 minutes at 30fps

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (25, 30, 40)
        self.COLOR_PLACEMENT_VALID = (0, 60, 20, 100)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_UI_BG = (50, 60, 70, 180)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game entities and state will be initialized in reset()
        self.base_health = 0
        self.max_base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.total_waves = 10
        self.game_over = False
        self.victory = False

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [0, 0]
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.steps = 0
        self.score = 0
        
        # Define Path and Placement Grid
        self._define_path_and_grid()
        self._define_tower_types()
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        # self.validate_implementation()

    def _define_path_and_grid(self):
        self.path_waypoints = [
            (0, 5), (3, 5), (3, 2), (7, 2), (7, 8),
            (11, 8), (11, 3), (14, 3), (14, 6), (16, 6)
        ]
        self.path_pixels = [(x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2) for x, y in self.path_waypoints]
        
        self.placement_grid = np.ones((self.GRID_W, self.GRID_H), dtype=bool)
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                        self.placement_grid[x, y] = False # Can't build on path
        
        # Also block around path
        path_coords = set()
        for i in range(len(self.path_waypoints) - 1):
            p1, p2 = self.path_waypoints[i], self.path_waypoints[i+1]
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2: # Vertical segment
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    path_coords.add((x1, y))
            else: # Horizontal segment
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    path_coords.add((x, y1))
        
        for x, y in path_coords:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                        self.placement_grid[nx, ny] = False


    def _define_tower_types(self):
        self.tower_types = [
            {
                "name": "Gatling", "cost": 25, "range": 100, "damage": 2, "cooldown": 10, "color": (70, 120, 255), "proj_speed": 15
            },
            {
                "name": "Cannon", "cost": 75, "range": 150, "damage": 10, "cooldown": 60, "color": (0, 80, 200), "proj_speed": 10
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.max_base_health = 50
        self.base_health = self.max_base_health
        self.resources = 100
        self.wave_number = 0
        self.wave_timer = 5 * self.FPS # 5 second countdown to first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.game_over = False
        self.victory = False
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.steps = 0
        self.score = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001 # Small penalty for time passing
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._calculate_reward()
            self._cleanup()

        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)

        # Place tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()

        # Switch tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
            # // SFX: ui_switch

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        x, y = self.cursor_pos
        tower_type = self.tower_types[self.selected_tower_idx]
        
        is_valid_placement = self.placement_grid[x, y]
        has_enough_resources = self.resources >= tower_type["cost"]
        
        if is_valid_placement and has_enough_resources:
            self.resources -= tower_type["cost"]
            new_tower = {
                "pos": (x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2),
                "type": tower_type,
                "cooldown_timer": 0,
                "target": None
            }
            self.towers.append(new_tower)
            self.placement_grid[x, y] = False # Can't build on top of another tower
            # // SFX: place_tower
            self._create_particles(new_tower["pos"], tower_type["color"], 20, 5, 20)
        else:
            # // SFX: error
            pass

    def _update_game_state(self):
        self._update_wave_manager()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
    def _update_wave_manager(self):
        if self.enemies_to_spawn == 0 and not self.enemies:
            if self.wave_number > 0 and not self.victory:
                 # Wave cleared bonus
                 self.score += 50
                 self.resources += 100 + self.wave_number * 10
                 if self.wave_number == self.total_waves:
                     self.victory = True
                     self.game_over = True
                     return

            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                self.wave_number += 1
                self.enemies_to_spawn = 5 + self.wave_number * 2
                self.spawn_timer = 0
                self.wave_timer = 10 * self.FPS # Time between waves
        
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.spawn_timer = self.FPS // 2 # Spawn every 0.5s

    def _spawn_enemy(self):
        difficulty_mod = 1.05 ** (self.wave_number - 1)
        new_enemy = {
            "pos": list(self.path_pixels[0]),
            "max_health": int(10 * difficulty_mod),
            "health": int(10 * difficulty_mod),
            "speed": 1.5 * difficulty_mod,
            "waypoint_idx": 1,
            "value": 5 + self.wave_number
        }
        self.enemies.append(new_enemy)

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown_timer"] > 0:
                tower["cooldown_timer"] -= 1
                continue

            # Find a target
            target = None
            min_dist = tower["type"]["range"] ** 2
            for enemy in self.enemies:
                dx = enemy["pos"][0] - tower["pos"][0]
                dy = enemy["pos"][1] - tower["pos"][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower["target"] = target
                tower["cooldown_timer"] = tower["type"]["cooldown"]
                self._fire_projectile(tower)
                # // SFX: tower_shoot

    def _fire_projectile(self, tower):
        self.projectiles.append({
            "pos": list(tower["pos"]),
            "target": tower["target"],
            "type": tower["type"]
        })
        self._create_particles(tower["pos"], (255, 255, 100), 5, 3, 10) # Muzzle flash

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["waypoint_idx"] >= len(self.path_pixels):
                continue # Already at the end

            target_pos = self.path_pixels[enemy["waypoint_idx"]]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["waypoint_idx"] += 1
                if enemy["waypoint_idx"] >= len(self.path_pixels):
                    # Reached base
                    self.base_health -= 5
                    self.score -= 5
                    # // SFX: base_damage
                    enemy["health"] = 0 # Mark for removal
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]

    def _update_projectiles(self):
        for proj in self.projectiles:
            if proj["target"]["health"] <= 0:
                proj["to_remove"] = True
                continue

            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            speed = proj["type"]["proj_speed"]

            if dist < speed:
                # Hit target
                proj["target"]["health"] -= proj["type"]["damage"]
                self.score += 0.1
                proj["to_remove"] = True
                self._create_particles(proj["pos"], self.COLOR_ENEMY, 10, 2, 15)
                # // SFX: enemy_hit
            else:
                proj["pos"][0] += (dx / dist) * speed
                proj["pos"][1] += (dy / dist) * speed
    
    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] = max(0, p['radius'] - 0.2)

    def _create_particles(self, pos, color, count, radius, life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(life // 2, life),
                'radius': random.uniform(radius * 0.5, radius),
                'color': color
            })

    def _calculate_reward(self):
        reward = 0
        for enemy in self.enemies:
            if enemy["health"] <= 0:
                reward += 1 # Kill bonus
                self.resources += enemy["value"]
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 30, 4, 30)
                # // SFX: enemy_death
        
        if self.base_health <= 0 and not self.game_over:
            reward -= 100 # Game over penalty
        
        if self.victory and not self.game_over:
            reward += 100 # Victory bonus
            
        return reward

    def _cleanup(self):
        new_enemies = []
        for enemy in self.enemies:
            if enemy["health"] > 0 and enemy["waypoint_idx"] < len(self.path_pixels):
                new_enemies.append(enemy)
        self.enemies = new_enemies
        
        self.projectiles = [p for p in self.projectiles if not p.get("to_remove", False)]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
        if self.victory:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "game_over": self.game_over,
            "victory": self.victory,
        }

    def _render_game(self):
        # Draw grid and placement areas
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                if self.placement_grid[x, y]:
                    s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_PLACEMENT_VALID)
                    self.screen.blit(s, rect[:2])
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw path
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, self.GRID_SIZE)

        # Draw base
        base_pos = self.path_pixels[-1]
        base_rect = (base_pos[0] - self.GRID_SIZE//2, base_pos[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, (200, 255, 200), base_rect, 2)
        
        # Draw towers
        for tower in self.towers:
            ttype = tower["type"]
            pos_int = (int(tower["pos"][0]), int(tower["pos"][1]))
            # Draw range circle
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], ttype["range"], (ttype["color"][0], ttype["color"][1], ttype["color"][2], 30))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], ttype["range"], ttype["color"])
            # Draw tower body
            tower_rect = (pos_int[0] - self.GRID_SIZE//3, pos_int[1] - self.GRID_SIZE//3, self.GRID_SIZE*2//3, self.GRID_SIZE*2//3)
            pygame.draw.rect(self.screen, ttype["color"], tower_rect)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, tower_rect, 2)

        # Draw enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            radius = self.GRID_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, (255, 100, 130))
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_w = radius * 2
            bar_h = 4
            pygame.draw.rect(self.screen, (50, 0, 0), (pos_int[0] - radius, pos_int[1] - radius - bar_h - 2, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos_int[0] - radius, pos_int[1] - radius - bar_h - 2, int(bar_w * health_pct), bar_h))

        # Draw projectiles
        for proj in self.projectiles:
            pos_int = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.circle(self.screen, (255, 255, 100), pos_int, 4)
            pygame.draw.circle(self.screen, (255, 255, 220), pos_int, 2)

        # Draw particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 30))
            color_with_alpha = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (pos_int[0] - p['radius'], pos_int[1] - p['radius']))

        # Draw cursor
        cursor_px_x, cursor_px_y = self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE
        cursor_rect = (cursor_px_x, cursor_px_y, self.GRID_SIZE, self.GRID_SIZE)
        
        tower_type = self.tower_types[self.selected_tower_idx]
        can_afford = self.resources >= tower_type["cost"]
        is_valid_spot = self.placement_grid[self.cursor_pos[0], self.cursor_pos[1]]
        
        cursor_color = (255, 255, 0) if can_afford and is_valid_spot else (255, 0, 0)
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 3)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, 0, self.WIDTH, 35)
        s = pygame.Surface((self.WIDTH, 35), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, 0))

        # Wave Info
        wave_text = f"Wave: {self.wave_number}/{self.total_waves}"
        if self.enemies_to_spawn == 0 and not self.enemies and not self.victory:
            wave_text += f" (Next in {self.wave_timer // self.FPS + 1}s)"
        text_surf = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 5))

        # Base Health Bar
        health_pct = self.base_health / self.max_base_health
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.WIDTH // 2 - bar_w // 2, 7
        pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, int(bar_w * health_pct), bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 2)
        health_label = self.font_small.render(f"{int(self.base_health)}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_label, (bar_x + bar_w/2 - health_label.get_width()/2, bar_y + 2))

        # Resources & Tower Info
        res_text = f"${self.resources}"
        res_surf = self.font_large.render(res_text, True, (255, 223, 0))
        self.screen.blit(res_surf, (self.WIDTH - res_surf.get_width() - 10, 5))

        tower_type = self.tower_types[self.selected_tower_idx]
        tower_info = f"Tower: {tower_type['name']} (${tower_type['cost']})"
        tower_surf = self.font_small.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_surf, (self.WIDTH - res_surf.get_width() - tower_surf.get_width() - 20, 10))
        
        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            
            text_surf = self.font_huge.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text_surf, text_rect)
            
            score_surf = self.font_large.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(score_surf, score_rect)

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

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    done = False
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)

    print(f"Game Over. Final Info: {info}")
    env.close()