
# Generated: 2025-08-28T05:16:20.348283
# Source Brief: brief_02573.md
# Brief Index: 2573

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. Space to place the selected turret. "
        "Shift to cycle between turret types."
    )

    game_description = (
        "A top-down tower defense game. Strategically place turrets to defend your base "
        "from waves of incoming enemies. Survive as long as you can!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and clock
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (25, 35, 45)
        self.COLOR_GRID = (20, 25, 35)
        self.COLOR_BASE = (0, 180, 100)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 50, 50)
        
        self.TURRET_SPECS = {
            1: {"name": "Gatling", "color": (0, 200, 255), "range": 80, "fire_rate": 5, "damage": 8, "cost": 1},
            2: {"name": "Cannon", "color": (255, 150, 0), "range": 120, "fire_rate": 20, "damage": 35, "cost": 1},
        }

        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 32)

        # Game constants
        self.MAX_STEPS = 1000
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_BASE_HEALTH = 100

        # Define enemy path waypoints
        self.path_waypoints = [
            (-20, 200), (100, 200), (100, 100), (400, 100),
            (400, 300), (200, 300), (200, 240), (self.WIDTH + 20, 240)
        ]
        self.base_pos = (self.WIDTH - 40, 240)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.enemies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []
        self.turret_resources = {}
        self.cursor_grid_pos = (0, 0)
        self.current_turret_type = 1
        self.wave_number = 0
        self.wave_spawn_timer = 0
        self.enemies_in_wave = 0
        self.last_action_buttons = [0, 0]
        self.reward_buffer = 0
        self.rng = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.MAX_BASE_HEALTH
        self.enemies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []
        self.turret_resources = {1: 15, 2: 5}
        self.cursor_grid_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.current_turret_type = 1
        self.wave_number = 0
        self.wave_spawn_timer = 0
        self.enemies_in_wave = 0
        self.last_action_buttons = [0, 0]
        self.reward_buffer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.reward_buffer = -0.01  # Time penalty

        self._handle_actions(action)
        self._update_spawner()
        self._update_turrets()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        self._cleanup_entities()

        self.steps += 1
        
        terminated = self.base_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            if self.base_health > 0:
                self.reward_buffer += 100  # Survival bonus
                self.score += 1000
            else:
                self.reward_buffer -= 100  # Base destroyed penalty
            self.game_over = True

        reward = self.reward_buffer
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_btn, shift_btn = action
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_grid_pos = (self.cursor_grid_pos[0], max(0, self.cursor_grid_pos[1] - 1))
        elif movement == 2: self.cursor_grid_pos = (self.cursor_grid_pos[0], min(self.GRID_H - 1, self.cursor_grid_pos[1] + 1))
        elif movement == 3: self.cursor_grid_pos = (max(0, self.cursor_grid_pos[0] - 1), self.cursor_grid_pos[1])
        elif movement == 4: self.cursor_grid_pos = (min(self.GRID_W - 1, self.cursor_grid_pos[0] + 1), self.cursor_grid_pos[1])

        # --- Place Turret (Space) ---
        if space_btn and not self.last_action_buttons[0]:
            if self._is_valid_placement(self.cursor_grid_pos):
                if self.turret_resources[self.current_turret_type] > 0:
                    # sfx: place_turret.wav
                    spec = self.TURRET_SPECS[self.current_turret_type]
                    pixel_pos = self._grid_to_pixel(self.cursor_grid_pos)
                    self.turrets.append({
                        "pos": pixel_pos,
                        "type": self.current_turret_type,
                        "range": spec["range"],
                        "fire_rate": spec["fire_rate"],
                        "cooldown": 0,
                    })
                    self.turret_resources[self.current_turret_type] -= spec["cost"]

        # --- Cycle Turret Type (Shift) ---
        if shift_btn and not self.last_action_buttons[1]:
            # sfx: ui_switch.wav
            self.current_turret_type = 2 if self.current_turret_type == 1 else 1

        self.last_action_buttons = [space_btn, shift_btn]

    def _grid_to_pixel(self, grid_pos):
        return (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, 
                grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)

    def _is_on_path(self, pixel_pos, margin=25):
        for i in range(len(self.path_waypoints) - 1):
            p1 = pygame.Vector2(self.path_waypoints[i])
            p2 = pygame.Vector2(self.path_waypoints[i+1])
            p3 = pygame.Vector2(pixel_pos)
            
            d = p2 - p1
            if d.length() == 0: continue
            t = max(0, min(1, (p3 - p1).dot(d) / d.length_squared()))
            closest_point = p1 + t * d
            if p3.distance_to(closest_point) < margin:
                return True
        return False

    def _is_valid_placement(self, grid_pos):
        pixel_pos = self._grid_to_pixel(grid_pos)
        if self._is_on_path(pixel_pos): return False
        for turret in self.turrets:
            if turret["pos"] == pixel_pos: return False
        return True

    def _update_spawner(self):
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.enemies_in_wave > 0:
            self.wave_spawn_timer = 15  # Spawn one enemy every 15 steps during a wave
            self.enemies_in_wave -= 1
            
            wave_level = self.wave_number // 2
            health = 30 + wave_level * 10
            speed = 1.0 + wave_level * 0.1
            
            self.enemies.append({
                "pos": list(self.path_waypoints[0]),
                "health": health,
                "max_health": health,
                "speed": speed,
                "path_index": 1,
            })
            
        elif len(self.enemies) == 0 and self.enemies_in_wave == 0:
            # Start next wave
            self.wave_number += 1
            self.enemies_in_wave = 5 + self.wave_number * 2
            self.wave_spawn_timer = 120 # Cooldown between waves

    def _update_turrets(self):
        for turret in self.turrets:
            turret["cooldown"] = max(0, turret["cooldown"] - 1)
            if turret["cooldown"] == 0:
                target = None
                min_dist = turret["range"]
                for enemy in self.enemies:
                    dist = math.hypot(enemy["pos"][0] - turret["pos"][0], enemy["pos"][1] - turret["pos"][1])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    # sfx: fire_gatling.wav or fire_cannon.wav
                    spec = self.TURRET_SPECS[turret["type"]]
                    self.projectiles.append({
                        "pos": list(turret["pos"]),
                        "target": target,
                        "speed": 10,
                        "damage": spec["damage"],
                        "color": spec["color"],
                        "dead": False
                    })
                    turret["cooldown"] = spec["fire_rate"]
                    # Muzzle flash
                    self.particles.append({
                        "pos": turret["pos"], "type": "flash", "radius": 8, "lifetime": 3, "color": (255, 255, 200)
                    })

    def _update_projectiles(self):
        for proj in self.projectiles:
            if proj["dead"]: continue
            
            target_enemy = proj["target"]
            if target_enemy["health"] <= 0:
                proj["dead"] = True
                continue

            target_pos = pygame.Vector2(target_enemy["pos"])
            proj_pos = pygame.Vector2(proj["pos"])
            
            direction = (target_pos - proj_pos)
            if direction.length() < proj["speed"]:
                # Hit target
                # sfx: hit_enemy.wav
                target_enemy["health"] -= proj["damage"]
                self.reward_buffer += 0.1
                proj["dead"] = True
                
                # Explosion particle
                self.particles.append({
                    "pos": target_enemy["pos"], "type": "explosion", "radius": 0, "max_radius": 15, "lifetime": 10, "color": (255, 200, 0)
                })

                if target_enemy["health"] <= 0:
                    # sfx: enemy_die.wav
                    self.score += 10
                    self.reward_buffer += 1.0
            else:
                direction.normalize_ip()
                proj_pos += direction * proj["speed"]
                proj["pos"] = [proj_pos.x, proj_pos.y]

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["path_index"] >= len(self.path_waypoints):
                # Reached base
                # sfx: base_damage.wav
                self.base_health -= 10
                enemy["health"] = 0 # Mark for deletion
                self.particles.append({
                    "pos": self.base_pos, "type": "base_hit", "radius": 0, "max_radius": 30, "lifetime": 15, "color": self.COLOR_BASE_DMG
                })
                continue
                
            target_pos = pygame.Vector2(self.path_waypoints[enemy["path_index"]])
            enemy_pos = pygame.Vector2(enemy["pos"])
            
            direction = target_pos - enemy_pos
            if direction.length() < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                direction.normalize_ip()
                enemy_pos += direction * enemy["speed"]
                enemy["pos"] = [enemy_pos.x, enemy_pos.y]
    
    def _update_particles(self):
        for p in self.particles:
            p["lifetime"] -= 1
            if p["type"] in ["explosion", "base_hit"]:
                p["radius"] += p["max_radius"] / p["lifetime"] if p["lifetime"] > 0 else p["max_radius"]
            elif p["type"] == "flash":
                p["radius"] *= 0.7

    def _cleanup_entities(self):
        self.enemies = [e for e in self.enemies if e["health"] > 0]
        self.projectiles = [p for p in self.projectiles if not p["dead"]]
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

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
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 40)
        
        # Draw base
        base_rect = pygame.Rect(self.base_pos[0] - 15, self.base_pos[1] - 15, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw turrets
        for turret in self.turrets:
            spec = self.TURRET_SPECS[turret["type"]]
            pos = (int(turret["pos"][0]), int(turret["pos"][1]))
            pygame.draw.rect(self.screen, spec["color"], (pos[0]-8, pos[1]-8, 16, 16))
            pygame.draw.rect(self.screen, self.COLOR_BG, (pos[0]-5, pos[1]-5, 10, 10))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-8, pos[1]-15, 16, 4))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0]-8, pos[1]-15, int(16 * health_pct), 4))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.draw.line(self.screen, proj["color"], pos, pos, 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["lifetime"] / 10)) if p["lifetime"] > 0 else 0
            if p["type"] in ["explosion", "base_hit"]:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), (*p["color"], alpha))
            elif p["type"] == "flash":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), (*p["color"], alpha))

        # Draw cursor
        cursor_pixel_pos = self._grid_to_pixel(self.cursor_grid_pos)
        is_valid = self._is_valid_placement(self.cursor_grid_pos)
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID
        
        # Draw range indicator
        spec = self.TURRET_SPECS[self.current_turret_type]
        pygame.gfxdraw.aacircle(self.screen, cursor_pixel_pos[0], cursor_pixel_pos[1], spec["range"], (*cursor_color, 60))
        
        # Draw cursor crosshair
        pygame.draw.line(self.screen, cursor_color, (cursor_pixel_pos[0] - 10, cursor_pixel_pos[1]), (cursor_pixel_pos[0] + 10, cursor_pixel_pos[1]), 1)
        pygame.draw.line(self.screen, cursor_color, (cursor_pixel_pos[0], cursor_pixel_pos[1] - 10), (cursor_pixel_pos[0], cursor_pixel_pos[1] + 10), 1)

    def _render_ui(self):
        # Base Health Bar
        health_pct = max(0, self.base_health / self.MAX_BASE_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render(f"Base: {int(self.base_health)}/{self.MAX_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score and Steps
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_small.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        # Turret selection and resources
        y_offset = self.HEIGHT - 30
        for i, t_type in enumerate(self.TURRET_SPECS.keys()):
            spec = self.TURRET_SPECS[t_type]
            color = spec["color"]
            is_selected = self.current_turret_type == t_type
            
            x_offset = 10 + i * 150
            
            # Selection indicator
            if is_selected:
                pygame.draw.rect(self.screen, color, (x_offset - 5, y_offset - 5, 130, 28), 2, border_radius=5)

            pygame.draw.rect(self.screen, color, (x_offset, y_offset, 16, 16))
            text = self.font_small.render(f"{spec['name']}: {self.turret_resources[t_type]}", True, self.COLOR_TEXT)
            self.screen.blit(text, (x_offset + 22, y_offset))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "turrets_placed": len(self.turrets),
            "enemies_on_screen": len(self.enemies),
            "wave": self.wave_number,
        }
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # To run and play the game manually
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' as appropriate
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    while running:
        action = [0, 0, 0] # no-op, release, release
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()