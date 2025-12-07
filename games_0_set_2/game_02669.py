
# Generated: 2025-08-28T05:33:16.070239
# Source Brief: brief_02669.md
# Brief Index: 2669

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑↓←→ to move the cursor. Press Space to place a tower. Press Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers on a grid in this minimalist, real-time tower defense game."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (32, 32, 32)
    COLOR_GRID = (48, 48, 48)
    COLOR_PATH = (64, 64, 64)
    COLOR_BASE = (76, 175, 80)
    COLOR_BASE_DAMAGE = (255, 100, 100)
    COLOR_ENEMY = (244, 67, 54)
    COLOR_TOWER_1 = (33, 150, 243)
    COLOR_TOWER_2 = (156, 39, 176)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_CURSOR = (255, 235, 59)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 40, 40)
    COLOR_HEALTH_BAR = (76, 175, 80)
    COLOR_HEALTH_BAR_BG = (90, 90, 90)

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    MAX_WAVES = 10
    
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 25, "range": 80, "damage": 5, "fire_rate": 10}, # Fires every 1/3 sec
        1: {"name": "Cannon", "cost": 75, "range": 150, "damage": 35, "fire_rate": 45} # Fires every 1.5 sec
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.path_waypoints = [
            (-self.CELL_SIZE, 3.5 * self.CELL_SIZE),
            (2.5 * self.CELL_SIZE, 3.5 * self.CELL_SIZE),
            (2.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (13.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (13.5 * self.CELL_SIZE, 1.5 * self.CELL_SIZE),
            (self.SCREEN_WIDTH + self.CELL_SIZE, 1.5 * self.CELL_SIZE),
        ]
        self.path_grid_coords = self._get_path_grid_coords()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 100
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.spawn_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.base_damage_flash = 0
        
        self.reset()
        self.validate_implementation()

    def _get_path_grid_coords(self):
        coords = set()
        for i in range(len(self.path_waypoints) - 1):
            p1 = np.array(self.path_waypoints[i])
            p2 = np.array(self.path_waypoints[i+1])
            dist = np.linalg.norm(p2 - p1)
            for t in np.arange(0, dist, self.CELL_SIZE / 4):
                point = p1 + (p2 - p1) * (t / dist)
                gx, gy = int(point[0] / self.CELL_SIZE), int(point[1] / self.CELL_SIZE)
                coords.add((gx, gy))
        return coords
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.max_base_health
        self.resources = 80
        self.current_wave = 0
        self.wave_timer = 150  # Delay before first wave
        self.spawn_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.base_damage_flash = 0
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return
        self.enemies_in_wave = 5 + self.current_wave * 2
        self.enemies_spawned = 0
        self.spawn_timer = 0
        self.wave_cleared_bonus_given = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.001 # Small time penalty

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Game Logic ---
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.enemies_spawned < self.enemies_in_wave:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.spawn_timer = max(10, 30 - self.current_wave)

        reward += self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1

        # Check for wave clear
        if not self.wave_cleared_bonus_given and self.enemies_spawned == self.enemies_in_wave and not self.enemies:
            reward += 10
            self.score += 10
            self.wave_cleared_bonus_given = True
            self.wave_timer = 200 # Time until next wave
            if self.current_wave < self.MAX_WAVES:
                self._start_next_wave()

        # --- Termination ---
        terminated = False
        if self.base_health <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        elif self.current_wave > self.MAX_WAVES and not self.enemies:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle Tower Type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_switch

        # Place Tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]:
            # sfx: error_buzz
            return

        gx, gy = self.cursor_pos
        is_on_path = (gx, gy) in self.path_grid_coords
        is_occupied = any(t['grid_pos'] == [gx, gy] for t in self.towers)

        if not is_on_path and not is_occupied:
            self.resources -= spec["cost"]
            self.towers.append({
                "type": self.selected_tower_type,
                "pos": ((gx + 0.5) * self.CELL_SIZE, (gy + 0.5) * self.CELL_SIZE),
                "grid_pos": [gx, gy],
                "cooldown": 0,
                "target": None
            })
            # sfx: place_tower
            
    def _spawn_enemy(self):
        self.enemies_spawned += 1
        speed = 1.0 + self.current_wave * 0.1
        health = 20 + self.current_wave * 10
        self.enemies.append({
            "pos": np.array(self.path_waypoints[0], dtype=float),
            "health": health,
            "max_health": health,
            "speed": speed,
            "waypoint_idx": 1
        })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            spec = self.TOWER_SPECS[tower['type']]
            
            # Find a target
            target = None
            for enemy in self.enemies:
                dist = math.dist(tower['pos'], enemy['pos'])
                if dist <= spec['range']:
                    target = enemy
                    break
            
            tower['target'] = target
            if target:
                self.projectiles.append({
                    "start_pos": np.array(tower['pos']),
                    "end_pos": np.array(target['pos']),
                    "pos": np.array(tower['pos']),
                    "damage": spec['damage'],
                    "speed": 15,
                    "type": tower['type']
                })
                tower['cooldown'] = spec['fire_rate']
                # sfx: fire_gatling or fire_cannon
        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            direction = proj['end_pos'] - proj['pos']
            dist = np.linalg.norm(direction)
            if dist < proj['speed']:
                self._handle_projectile_impact(proj)
                self.projectiles.remove(proj)
                continue
            
            proj['pos'] += (direction / dist) * proj['speed']

    def _handle_projectile_impact(self, proj):
        # Create visual effect
        for _ in range(5):
            self.particles.append(self._create_particle(proj['pos'], self.COLOR_PROJECTILE, 2, 8))
        
        # Check for hit
        spec = self.TOWER_SPECS[proj['type']]
        hit_radius = 10 if proj['type'] == 0 else 20 # Gatling vs Cannon splash
        
        for enemy in self.enemies:
            if math.dist(proj['pos'], enemy['pos']) < hit_radius + 10: # 10 is enemy radius
                 enemy['health'] -= proj['damage']
                 # sfx: enemy_hit
                 # Small reward for hitting
                 self.score += 0.1
                 # No direct reward in step, it is calculated from enemy death

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                self.base_health -= 10
                self.base_damage_flash = 10
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            target_pos = np.array(self.path_waypoints[enemy['waypoint_idx']])
            direction = target_pos - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['waypoint_idx'] += 1
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']

            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.resources += 5 + self.current_wave
                reward += 1 # Reward for kill
                # sfx: enemy_explode
                for _ in range(20):
                    self.particles.append(self._create_particle(enemy['pos'], self.COLOR_ENEMY, 3, 15))
        return reward

    def _create_particle(self, pos, color, speed, lifespan):
        angle = random.uniform(0, 2 * math.pi)
        vel = [math.cos(angle) * speed * random.uniform(0.5, 1.5), 
               math.sin(angle) * speed * random.uniform(0.5, 1.5)]
        return {"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color}
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

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
            "wave": self.current_wave,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.CELL_SIZE)

        # Draw base
        base_color = self.COLOR_BASE_DAMAGE if self.base_damage_flash > 0 else self.COLOR_BASE
        base_rect = pygame.Rect(self.SCREEN_WIDTH - self.CELL_SIZE, 1 * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, base_color, base_rect)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*base_color, 150))


        # Draw towers and their ranges (if cursor is on them)
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            color = self.COLOR_TOWER_1 if tower['type'] == 0 else self.COLOR_TOWER_2
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            
            if tower['grid_pos'] == self.cursor_pos:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec['range'], (*color, 30))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec['range'], (*color, 100))
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 16, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 16, (255,255,255,50))
            
            # Cooldown indicator
            cooldown_ratio = tower['cooldown'] / spec['fire_rate']
            if cooldown_ratio > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(16 * cooldown_ratio), (*self.COLOR_BG, 150))


        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, (0,0,0,100))
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 20
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_width/2, pos[1] - 18, bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_width/2, pos[1] - 18, bar_width * health_ratio, 4))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 15))
            color = (*p['color'], alpha)
            size = int(p['lifespan'] / 4)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Draw cursor
        self._render_cursor()

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        color = self.COLOR_TOWER_1 if self.selected_tower_type == 0 else self.COLOR_TOWER_2
        
        # Determine placement validity
        is_on_path = (gx, gy) in self.path_grid_coords
        is_occupied = any(t['grid_pos'] == [gx, gy] for t in self.towers)
        has_funds = self.resources >= spec['cost']
        is_valid = not is_on_path and not is_occupied and has_funds
        
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_ENEMY
        
        # Draw range indicator
        center_x = int((gx + 0.5) * self.CELL_SIZE)
        center_y = int((gy + 0.5) * self.CELL_SIZE)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, spec['range'], (*cursor_color, 20))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, spec['range'], (*cursor_color, 80))
        
        # Draw cursor box
        rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, (*cursor_color, 100), rect, 0)
        pygame.draw.rect(self.screen, cursor_color, rect, 2)

    def _render_ui(self):
        # Top-left info panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, 180, 60))
        wave_text = self.font_small.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(resource_text, (10, 35))

        # Bottom health bar
        health_ratio = self.base_health / self.max_base_health
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, self.SCREEN_HEIGHT - 25, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, self.SCREEN_HEIGHT - 25, bar_width * health_ratio, 15))
        health_text = self.font_small.render(f"Base Health: {self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_width / 2 + 10 - health_text.get_width()/2, self.SCREEN_HEIGHT - 26))

        # Bottom-right tower info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        color = self.COLOR_TOWER_1 if self.selected_tower_type == 0 else self.COLOR_TOWER_2
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 180, self.SCREEN_HEIGHT - 80, 180, 80))
        name_text = self.font_small.render(f"Type: {spec['name']}", True, color)
        cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_TEXT)
        dmg_text = self.font_small.render(f"Damage: {spec['damage']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (self.SCREEN_WIDTH - 170, self.SCREEN_HEIGHT - 70))
        self.screen.blit(cost_text, (self.SCREEN_WIDTH - 170, self.SCREEN_HEIGHT - 50))
        self.screen.blit(dmg_text, (self.SCREEN_WIDTH - 170, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.base_health <= 0:
                msg = "GAME OVER"
                msg_color = self.COLOR_ENEMY
            else:
                msg = "YOU WIN!"
                msg_color = self.COLOR_BASE
            
            text_surf = self.font_large.render(msg, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)
            
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' as appropriate

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
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
        total_reward += reward

        # --- Pygame rendering ---
        # The observation is already the rendered screen, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(30)


        clock.tick(30) # Limit to 30 FPS

    env.close()