
# Generated: 2025-08-27T14:44:25.045370
# Source Brief: brief_00778.md
# Brief Index: 778

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press space to build a high-damage Cannon Tower. "
        "Hold shift to build a long-range Laser Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a grid. "
        "Survive all 10 waves to win."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    MAX_STEPS = 30 * 60 * 2 # 2 minutes at 30fps
    TOTAL_WAVES = 10

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_PATH = (45, 60, 90)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (200, 50, 50)
    
    COLOR_ENEMY = (217, 30, 24)
    COLOR_ENEMY_HP_BG = (80, 10, 8)
    COLOR_ENEMY_HP = (150, 20, 15)

    COLOR_TOWER_1 = (0, 150, 255) # Cannon
    COLOR_TOWER_2 = (255, 0, 150) # Laser
    COLOR_PROJECTILE_1 = (100, 200, 255)
    COLOR_PROJECTILE_2 = (255, 100, 200)

    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (230, 230, 230)
    COLOR_UI_BG = (25, 30, 45)
    
    # --- Tower Stats ---
    TOWER_SPECS = {
        1: {"cost": 100, "range": 2.5 * CELL_SIZE, "damage": 25, "fire_rate": 30}, # Cannon
        2: {"cost": 150, "range": 4.5 * CELL_SIZE, "damage": 10, "fire_rate": 10}, # Laser
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.game_objects = []
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.path_waypoints = []
        self.occupied_cells = set()
        
        self.reset()
        
        # This check is for development and ensures the implementation matches the spec
        # self.validate_implementation()

    def _generate_path(self):
        self.path_waypoints = []
        path_grid = []
        
        # Start on the left edge, vertically centered
        start_y = self.np_random.integers(3, self.GRID_HEIGHT - 3)
        current_pos = [0, start_y]
        path_grid.append(tuple(current_pos))

        # Create a horizontal path with vertical zig-zags
        direction = 1 # 1 for right
        while current_pos[0] < self.GRID_WIDTH - 2:
            # Move horizontally
            move_len = self.np_random.integers(3, 6)
            for _ in range(move_len):
                if current_pos[0] < self.GRID_WIDTH - 2:
                    current_pos[0] += direction
                    path_grid.append(tuple(current_pos))
            
            # Move vertically
            v_direction = 1 if current_pos[1] < self.GRID_HEIGHT / 2 else -1
            v_len = self.np_random.integers(2, 4)
            for _ in range(v_len):
                new_y = current_pos[1] + v_direction
                if 0 <= new_y < self.GRID_HEIGHT:
                    current_pos[1] = new_y
                    path_grid.append(tuple(current_pos))

        # Final stretch to the base on the right edge
        self.base_pos_grid = (self.GRID_WIDTH - 1, current_pos[1])
        while current_pos[0] < self.GRID_WIDTH -1:
            current_pos[0] += 1
            path_grid.append(tuple(current_pos))

        # Convert grid coordinates to pixel coordinates
        self.path_waypoints = [((x + 0.5) * self.CELL_SIZE, (y + 0.5) * self.CELL_SIZE) for x, y in path_grid]
        return set(path_grid)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.game_over_timer = 0
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 250
        
        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned = 0

        self.towers = []
        self.enemies = []
        self.projectiles = []

        self.path_cells = self._generate_path()
        self.occupied_cells = set(self.path_cells)
        self.occupied_cells.add(self.base_pos_grid)
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        self.current_wave += 1
        self.enemies_spawned = 0
        self.enemies_in_wave = 5 + (self.current_wave - 1) * 2
        
        self.base_enemy_health = 50 * (1 + 0.10 * (self.current_wave - 1))
        self.base_enemy_speed = 0.8 * (1 + 0.05 * (self.current_wave - 1))

    def step(self, action):
        movement, place_tower1, place_tower2 = action[0], action[1] == 1, action[2] == 1
        reward = 0

        if not self.game_over:
            # --- Player Actions ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1) # Down
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1) # Right
            
            cursor_tuple = tuple(self.cursor_pos)
            if cursor_tuple not in self.occupied_cells:
                if place_tower1 and self.resources >= self.TOWER_SPECS[1]["cost"]:
                    self._place_tower(1, cursor_tuple)
                    # sfx: place_tower_1.wav
                elif place_tower2 and self.resources >= self.TOWER_SPECS[2]["cost"]:
                    self._place_tower(2, cursor_tuple)
                    # sfx: place_tower_2.wav

            # --- Game Logic Update ---
            self._update_towers()
            self._update_projectiles()
            reward += self._update_enemies()
            
            # --- Wave Management ---
            if not self.enemies and self.enemies_spawned == self.enemies_in_wave:
                if self.current_wave > 0 and self.current_wave < self.TOTAL_WAVES:
                    reward += 1.0 # Wave complete bonus
                    self.resources += 100 + self.current_wave * 25
                
                self.wave_timer = 150 # Cooldown between waves
                if self.current_wave < self.TOTAL_WAVES:
                    self._spawn_wave()
                elif self.current_wave >= self.TOTAL_WAVES and not self.game_over:
                    self.victory = True
                    self.game_over = True
                    reward += 100
            
            self.wave_timer = max(0, self.wave_timer - 1)
            if self.wave_timer == 0 and self.enemies_spawned < self.enemies_in_wave:
                self._spawn_enemy()
                self.wave_timer = self.np_random.integers(15, 30) # Time between enemies in a wave

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            self.victory = False
            reward -= 100
            # sfx: game_over.wav
        
        if self.game_over:
            self.game_over_timer += 1
        
        if self.game_over_timer > 90: # 3 seconds
             terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over:
                reward -= 50 # Penalty for timeout
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self, tower_type, pos):
        spec = self.TOWER_SPECS[tower_type]
        self.resources -= spec["cost"]
        self.towers.append({
            "type": tower_type,
            "pos_grid": pos,
            "pos_px": ((pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE),
            "cooldown": 0,
            "spec": spec
        })
        self.occupied_cells.add(pos)

    def _spawn_enemy(self):
        self.enemies_spawned += 1
        self.enemies.append({
            "pos_px": list(self.path_waypoints[0]),
            "path_index": 0,
            "health": self.base_enemy_health,
            "max_health": self.base_enemy_health,
            "speed": self.base_enemy_speed * self.np_random.uniform(0.9, 1.1),
            "distance_traveled": 0.0
        })

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = None
                # Find the enemy that is furthest along the path
                best_dist = -1
                for enemy in self.enemies:
                    dist_sq = (enemy["pos_px"][0] - tower["pos_px"][0])**2 + (enemy["pos_px"][1] - tower["pos_px"][1])**2
                    if dist_sq < tower["spec"]["range"]**2:
                        if enemy["distance_traveled"] > best_dist:
                            best_dist = enemy["distance_traveled"]
                            target = enemy
                
                if target:
                    tower["cooldown"] = tower["spec"]["fire_rate"]
                    self.projectiles.append({
                        "type": tower["type"],
                        "pos_px": list(tower["pos_px"]),
                        "target": target,
                        "damage": tower["spec"]["damage"]
                    })
                    # sfx: fire_tower_1.wav or fire_tower_2.wav

    def _update_projectiles(self):
        projectiles_to_remove = []
        for proj in self.projectiles:
            if proj["target"] not in self.enemies: # Target already dead
                projectiles_to_remove.append(proj)
                continue

            target_pos = proj["target"]["pos_px"]
            dx = target_pos[0] - proj["pos_px"][0]
            dy = target_pos[1] - proj["pos_px"][1]
            dist = math.hypot(dx, dy)
            
            speed = 12
            if dist < speed:
                proj["target"]["health"] -= proj["damage"]
                projectiles_to_remove.append(proj)
                # sfx: hit_enemy.wav
            else:
                proj["pos_px"][0] += (dx / dist) * speed
                proj["pos_px"][1] += (dy / dist) * speed
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]

    def _update_enemies(self):
        enemies_to_remove = []
        reward = 0
        for enemy in self.enemies:
            if enemy["health"] <= 0:
                enemies_to_remove.append(enemy)
                reward += 0.1 # Kill reward
                self.score += 10
                self.resources += 15
                # sfx: enemy_die.wav
                continue

            # Move along path
            if enemy["path_index"] < len(self.path_waypoints) - 1:
                target_waypoint = self.path_waypoints[enemy["path_index"] + 1]
                dx = target_waypoint[0] - enemy["pos_px"][0]
                dy = target_waypoint[1] - enemy["pos_px"][1]
                dist = math.hypot(dx, dy)

                if dist < enemy["speed"]:
                    enemy["path_index"] += 1
                    enemy["distance_traveled"] += dist
                else:
                    move_x = (dx / dist) * enemy["speed"]
                    move_y = (dy / dist) * enemy["speed"]
                    enemy["pos_px"][0] += move_x
                    enemy["pos_px"][1] += move_y
                    enemy["distance_traveled"] += enemy["speed"]
            else: # Reached the base
                self.base_health = max(0, self.base_health - 20)
                enemies_to_remove.append(enemy)
                # sfx: base_damage.wav
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_path(self):
        for x, y in self.path_cells:
            pygame.draw.rect(self.screen, self.COLOR_PATH, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

    def _render_base(self):
        base_rect = pygame.Rect(self.base_pos_grid[0] * self.CELL_SIZE, self.base_pos_grid[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        health_ratio = self.base_health / self.max_base_health
        color = tuple(int(c1 * health_ratio + c2 * (1 - health_ratio)) for c1, c2 in zip(self.COLOR_BASE, self.COLOR_BASE_DMG))
        pygame.draw.rect(self.screen, color, base_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID, base_rect, 2)

    def _render_towers(self):
        for tower in self.towers:
            pos = (int(tower["pos_px"][0]), int(tower["pos_px"][1]))
            color = self.COLOR_TOWER_1 if tower["type"] == 1 else self.COLOR_TOWER_2
            
            # Draw range circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(tower["spec"]["range"]), (*color, 50))
            
            if tower["type"] == 1: # Cannon (Square)
                size = self.CELL_SIZE * 0.7
                rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2)
            else: # Laser (Triangle)
                size = self.CELL_SIZE * 0.4
                points = [
                    (pos[0], pos[1] - size),
                    (pos[0] - size, pos[1] + size * 0.7),
                    (pos[0] + size, pos[1] + size * 0.7)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos_px"][0]), int(enemy["pos_px"][1]))
            radius = self.CELL_SIZE * 0.3
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), self.COLOR_ENEMY)

            # Health bar
            hp_ratio = max(0, enemy["health"] / enemy["max_health"])
            bar_width = self.CELL_SIZE * 0.8
            bar_height = 4
            bar_y = pos[1] - radius - 8
            bg_rect = pygame.Rect(pos[0] - bar_width/2, bar_y, bar_width, bar_height)
            hp_rect = pygame.Rect(pos[0] - bar_width/2, bar_y, bar_width * hp_ratio, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HP_BG, bg_rect)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HP, hp_rect)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos_px"][0]), int(proj["pos_px"][1]))
            color = self.COLOR_PROJECTILE_1 if proj["type"] == 1 else self.COLOR_PROJECTILE_2
            size = 4 if proj["type"] == 1 else 2
            pygame.draw.rect(self.screen, color, (pos[0]-size/2, pos[1]-size/2, size, size))

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        
        is_valid_spot = tuple(self.cursor_pos) not in self.occupied_cells
        
        if is_valid_spot:
            s.fill((*self.COLOR_CURSOR, 60)) # Semi-transparent fill
        else:
            s.fill((*self.COLOR_ENEMY, 60))

        self.screen.blit(s, (x * self.CELL_SIZE, y * self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        # Top-left: Wave info
        wave_text = f"Wave: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.wave_timer > 0 and self.current_wave > 0 and not self.enemies:
            wave_text += f" (Next in {self.wave_timer//30 + 1}s)"
        
        self._draw_text(wave_text, (10, 10), self.font_small)

        # Top-right: Base Health
        self._draw_text(f"Base HP: {self.base_health}", (self.SCREEN_WIDTH - 150, 10), self.font_small)
        
        # Bottom-left: Resources
        self._draw_text(f"Resources: ${self.resources}", (10, self.SCREEN_HEIGHT - 30), self.font_small)
        
        # Bottom-right: Score
        self._draw_text(f"Score: {self.score}", (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 30), self.font_small)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        text = "VICTORY!" if self.victory else "GAME OVER"
        text_surf = self.font_large.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # Use a simple human-in-the-loop controller
    # This requires pygame to be initialized with a video driver.
    # If running headless, this part should be commented out.
    import os
    if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
        pygame.display.set_caption("Tower Defense")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        # Human controls
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

        action = np.array([movement, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Wave: {info['wave']}")

        # Render to the screen
        if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit to 30 FPS

    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")